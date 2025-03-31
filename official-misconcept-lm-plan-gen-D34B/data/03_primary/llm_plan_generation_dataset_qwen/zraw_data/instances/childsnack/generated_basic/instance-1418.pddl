; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 73868

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child141 child148 - child
    bread142 bread378 - bread-portion
    content0 content339 - content-portion
    tray76 - tray
    table1 table460 table386 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray76 kitchen)
     (at_kitchen_bread bread142)
     (at_kitchen_bread bread378)
     (at_kitchen_content content0)
     (at_kitchen_content content339)
     (no_gluten_bread bread378)
     (no_gluten_content content0)
     (allergic_gluten child148)
     (not_allergic_gluten child141)
     (waiting child141 table386)
     (waiting child148 table460)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child141)
     (served child148)
    )
  )
)
