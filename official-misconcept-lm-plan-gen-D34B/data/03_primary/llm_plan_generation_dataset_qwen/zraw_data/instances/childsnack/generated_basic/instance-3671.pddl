; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 363760

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child339 child183 - child
    bread286 bread330 - bread-portion
    content305 content53 - content-portion
    tray26 tray371 - tray
    table158 table102 table436 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray26 kitchen)
     (at tray371 kitchen)
     (at_kitchen_bread bread286)
     (at_kitchen_bread bread330)
     (at_kitchen_content content305)
     (at_kitchen_content content53)
     (no_gluten_bread bread330)
     (no_gluten_content content53)
     (allergic_gluten child183)
     (not_allergic_gluten child339)
     (waiting child339 table436)
     (waiting child183 table436)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child339)
     (served child183)
    )
  )
)
