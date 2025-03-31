; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 284936

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child138 - child
    bread207 - bread-portion
    content314 - content-portion
    tray485 tray63 - tray
    table95 table274 table344 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray485 kitchen)
     (at tray63 kitchen)
     (at_kitchen_bread bread207)
     (at_kitchen_content content314)
     (not_allergic_gluten child138)
     (waiting child138 table344)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child138)
    )
  )
)
