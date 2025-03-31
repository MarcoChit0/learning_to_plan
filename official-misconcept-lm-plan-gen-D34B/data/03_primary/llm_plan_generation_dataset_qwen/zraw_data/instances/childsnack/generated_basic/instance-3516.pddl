; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 793127

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child35 child24 - child
    bread350 bread252 - bread-portion
    content385 content194 - content-portion
    tray57 tray62 - tray
    table13 table236 table260 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray57 kitchen)
     (at tray62 kitchen)
     (at_kitchen_bread bread350)
     (at_kitchen_bread bread252)
     (at_kitchen_content content385)
     (at_kitchen_content content194)
     (not_allergic_gluten child24)
     (not_allergic_gluten child35)
     (waiting child35 table260)
     (waiting child24 table13)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child35)
     (served child24)
    )
  )
)
