; child-snack task with 2 children and 0.53 gluten factor 
; constant factor of 1.9
; random seed: 978395

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child472 child211 - child
    bread272 bread269 - bread-portion
    content296 content110 - content-portion
    tray352 tray263 tray306 - tray
    table291 table251 table340 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray352 kitchen)
     (at tray263 kitchen)
     (at tray306 kitchen)
     (at_kitchen_bread bread272)
     (at_kitchen_bread bread269)
     (at_kitchen_content content296)
     (at_kitchen_content content110)
     (no_gluten_bread bread269)
     (no_gluten_content content110)
     (allergic_gluten child211)
     (not_allergic_gluten child472)
     (waiting child472 table291)
     (waiting child211 table251)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child472)
     (served child211)
    )
  )
)
