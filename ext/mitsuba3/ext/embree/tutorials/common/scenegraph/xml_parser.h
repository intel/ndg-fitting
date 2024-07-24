// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../default.h"

namespace embree
{
  /* an XML node */
  class XML : public RefCount
  {
  public:
    XML (const std::string& name = "") : name(name) {}

    /*! returns number of children of XML node */
    size_t size() const { return children.size(); }

    /*! checks if child with specified node tag exists */
    bool hasChild(const std::string& childID) const 
    {
      for (size_t i=0; i<children.size(); i++)
        if (children[i]->name == childID) return true;
      return false;
    }

    /*! returns a parameter of the XML node */
    std::string parm(const std::string& parmID) const {
      std::map<std::string,std::string>::const_iterator i = parms.find(parmID);
      if (i == parms.end()) return ""; else return i->second;
    }

    Vec2f parm_Vec2f(const std::string& parmID) const {
      std::map<std::string,std::string>::const_iterator i = parms.find(parmID);
      if (i == parms.end()) THROW_RUNTIME_ERROR (loc.str()+": XML node has no parameter \"" + parmID + "\"");
      return string_to_Vec2f(i->second);
    }

    Vec3fa parm_Vec3fa(const std::string& parmID) const {
      std::map<std::string,std::string>::const_iterator i = parms.find(parmID);
      if (i == parms.end()) THROW_RUNTIME_ERROR (loc.str()+": XML node has no parameter \"" + parmID + "\"");
      return Vec3fa(string_to_Vec3f(i->second));
    }

    float parm_float(const std::string& parmID) const {
      std::map<std::string,std::string>::const_iterator i = parms.find(parmID);
      if (i == parms.end()) THROW_RUNTIME_ERROR (loc.str()+": XML node has no parameter \"" + parmID + "\"");
      return std::stof(i->second);
    }

    /*! returns the nth child */
    const Ref<XML> child(const size_t id) const 
    {
      if (id >= children.size()) 
        THROW_RUNTIME_ERROR (loc.str()+": XML node has no child \"" + toString(id) + "\"");
      return children[id];
    }

    /*! returns child by node tag */
    const Ref<XML> child(const std::string& childID) const 
    {
      for (size_t i=0; i<children.size(); i++)
        if (children[i]->name == childID) return children[i];
      THROW_RUNTIME_ERROR (loc.str()+": XML node has no child \"" + childID + "\"");
    }

    /*! returns child by node tag without failing */
    const Ref<XML> childOpt(const std::string& childID) const 
    {
      for (size_t i=0; i<children.size(); i++)
        if (children[i]->name == childID) return children[i];
      return null;
    }

    /*! adds a new parameter to the node */
    Ref<XML> add(const std::string& name, const std::string& val) {
      parms[name] = val; 
      return this; 
    }

    /*! adds a new child */
    Ref<XML> add(const Ref<XML>& xml) { 
      children.push_back(xml); 
      return this; 
    }

    /*! adds new data tokens to the body of the node */
    Ref<XML> add(const Token& tok) {
      body.push_back(tok);     
      return this; 
    }

    /*! compares two XML nodes */
    friend bool operator ==( const Ref<XML>& a, const Ref<XML>& b ) {
      return a->name == b->name && a->parms == b->parms && a->children == b->children && a->body == b->body;
    }

    /*! orders two XML nodes */
    friend bool operator <( const Ref<XML>& a, const Ref<XML>& b ) {
      if (a->name     != b->name    ) return a->name     < b->name;
      if (a->parms    != b->parms   ) return a->parms    < b->parms;
      if (a->children != b->children) return a->children < b->children;
      if (a->body     != b->body    ) return a->body     < b->body;
      return false;
    }

  public:
    ParseLocation loc;
    std::string name;
    std::map<std::string,std::string> parms;
    std::vector<Ref<XML> > children;
    std::vector<Token> body;
  };

  /*! load XML file from stream */
  std::istream& operator>>(std::istream& cin, Ref<XML>& xml);

  /*! load XML file from disk */
  Ref<XML> parseXML(const FileName& fileName, std::string id = "", bool hasHeader = true);

  /* store XML to stream */
  std::ostream& operator<<(std::ostream& cout, const Ref<XML>& xml);

  /*! store XML to disk */
  void emitXML(const FileName& fileName, const Ref<XML>& xml);
}

