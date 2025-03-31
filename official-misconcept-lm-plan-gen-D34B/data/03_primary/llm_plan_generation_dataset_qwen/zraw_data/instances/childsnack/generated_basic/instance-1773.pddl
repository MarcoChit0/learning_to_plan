; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 782861

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child148 - child
    bread131 - bread-portion
    content204 - content-portion
    tray63 - tray
    table107 table36 table448 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray63 kitchen)
     (at_kitchen_bread bread131)
     (at_kitchen_content content204)
     (not_allergic_gluten child148)
     (waiting child148 table36)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child148)
    )
  )
)
