; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 569859

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child123 child324 - child
    bread250 bread291 - bread-portion
    content114 content40 - content-portion
    tray484 - tray
    table163 table459 table9 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray484 kitchen)
     (at_kitchen_bread bread250)
     (at_kitchen_bread bread291)
     (at_kitchen_content content114)
     (at_kitchen_content content40)
     (no_gluten_bread bread291)
     (no_gluten_content content114)
     (allergic_gluten child123)
     (not_allergic_gluten child324)
     (waiting child123 table163)
     (waiting child324 table163)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child123)
     (served child324)
    )
  )
)
