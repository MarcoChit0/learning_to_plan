; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 394856

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child228 - child
    bread55 - bread-portion
    content339 - content-portion
    tray87 tray389 tray197 tray329 - tray
    table485 table398 table296 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray87 kitchen)
     (at tray389 kitchen)
     (at tray197 kitchen)
     (at tray329 kitchen)
     (at_kitchen_bread bread55)
     (at_kitchen_content content339)
     (not_allergic_gluten child228)
     (waiting child228 table398)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child228)
    )
  )
)
