; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 326914

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child403 child183 - child
    bread376 bread158 - bread-portion
    content333 content86 - content-portion
    tray368 tray462 - tray
    table453 table286 table345 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray368 kitchen)
     (at tray462 kitchen)
     (at_kitchen_bread bread376)
     (at_kitchen_bread bread158)
     (at_kitchen_content content333)
     (at_kitchen_content content86)
     (no_gluten_bread bread158)
     (no_gluten_content content86)
     (allergic_gluten child183)
     (not_allergic_gluten child403)
     (waiting child403 table453)
     (waiting child183 table453)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child403)
     (served child183)
    )
  )
)
