; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 766462

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child419 - child
    bread17 - bread-portion
    content200 - content-portion
    tray79 tray469 - tray
    table338 table173 table28 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray79 kitchen)
     (at tray469 kitchen)
     (at_kitchen_bread bread17)
     (at_kitchen_content content200)
     (not_allergic_gluten child419)
     (waiting child419 table28)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child419)
    )
  )
)
