; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 588049

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child391 child135 - child
    bread397 bread172 - bread-portion
    content33 content214 - content-portion
    tray488 tray357 tray124 - tray
    table156 table416 table74 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray488 kitchen)
     (at tray357 kitchen)
     (at tray124 kitchen)
     (at_kitchen_bread bread397)
     (at_kitchen_bread bread172)
     (at_kitchen_content content33)
     (at_kitchen_content content214)
     (no_gluten_bread bread397)
     (no_gluten_content content33)
     (allergic_gluten child135)
     (not_allergic_gluten child391)
     (waiting child391 table156)
     (waiting child135 table416)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child391)
     (served child135)
    )
  )
)
