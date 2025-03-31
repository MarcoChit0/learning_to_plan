; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 488871

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child30 - child
    bread200 - bread-portion
    content344 - content-portion
    tray345 tray435 tray29 - tray
    table62 table414 table342 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray345 kitchen)
     (at tray435 kitchen)
     (at tray29 kitchen)
     (at_kitchen_bread bread200)
     (at_kitchen_content content344)
     (not_allergic_gluten child30)
     (waiting child30 table62)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child30)
    )
  )
)
