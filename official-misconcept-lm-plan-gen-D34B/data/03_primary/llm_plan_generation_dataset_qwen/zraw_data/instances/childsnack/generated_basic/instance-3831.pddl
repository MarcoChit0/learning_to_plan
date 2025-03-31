; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 315442

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child70 - child
    bread97 - bread-portion
    content411 - content-portion
    tray41 tray448 tray343 tray134 - tray
    table165 table308 table218 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray41 kitchen)
     (at tray448 kitchen)
     (at tray343 kitchen)
     (at tray134 kitchen)
     (at_kitchen_bread bread97)
     (at_kitchen_content content411)
     (not_allergic_gluten child70)
     (waiting child70 table218)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child70)
    )
  )
)
