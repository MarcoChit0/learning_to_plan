; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 613989

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child493 child454 - child
    bread218 bread203 - bread-portion
    content233 content327 - content-portion
    tray444 tray272 tray213 - tray
    table325 table38 table304 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray444 kitchen)
     (at tray272 kitchen)
     (at tray213 kitchen)
     (at_kitchen_bread bread218)
     (at_kitchen_bread bread203)
     (at_kitchen_content content233)
     (at_kitchen_content content327)
     (not_allergic_gluten child454)
     (not_allergic_gluten child493)
     (waiting child493 table325)
     (waiting child454 table38)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child493)
     (served child454)
    )
  )
)
