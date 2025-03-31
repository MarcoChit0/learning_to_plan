; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 925666

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child463 - child
    bread470 - bread-portion
    content369 - content-portion
    tray314 tray168 tray67 tray89 - tray
    table454 table157 table495 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray314 kitchen)
     (at tray168 kitchen)
     (at tray67 kitchen)
     (at tray89 kitchen)
     (at_kitchen_bread bread470)
     (at_kitchen_content content369)
     (not_allergic_gluten child463)
     (waiting child463 table157)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child463)
    )
  )
)
