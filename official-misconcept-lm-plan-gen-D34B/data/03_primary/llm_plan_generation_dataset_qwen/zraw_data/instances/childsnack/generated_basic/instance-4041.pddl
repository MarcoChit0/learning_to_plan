; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 505111

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child208 child87 child42 - child
    bread221 bread249 bread218 - bread-portion
    content417 content113 content396 - content-portion
    tray457 - tray
    table407 table149 table164 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray457 kitchen)
     (at_kitchen_bread bread221)
     (at_kitchen_bread bread249)
     (at_kitchen_bread bread218)
     (at_kitchen_content content417)
     (at_kitchen_content content113)
     (at_kitchen_content content396)
     (not_allergic_gluten child42)
     (not_allergic_gluten child87)
     (not_allergic_gluten child208)
     (waiting child208 table149)
     (waiting child87 table164)
     (waiting child42 table164)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child208)
     (served child87)
     (served child42)
    )
  )
)
