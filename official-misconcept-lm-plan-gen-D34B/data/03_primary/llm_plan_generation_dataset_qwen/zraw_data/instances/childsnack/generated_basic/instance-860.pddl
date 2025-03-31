; child-snack task with 2 children and 0.59 gluten factor 
; constant factor of 1.9
; random seed: 331110

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child442 child154 - child
    bread254 bread327 - bread-portion
    content162 content375 - content-portion
    tray149 tray25 - tray
    table121 table102 table294 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray149 kitchen)
     (at tray25 kitchen)
     (at_kitchen_bread bread254)
     (at_kitchen_bread bread327)
     (at_kitchen_content content162)
     (at_kitchen_content content375)
     (no_gluten_bread bread327)
     (no_gluten_content content375)
     (allergic_gluten child154)
     (not_allergic_gluten child442)
     (waiting child442 table294)
     (waiting child154 table294)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child442)
     (served child154)
    )
  )
)
