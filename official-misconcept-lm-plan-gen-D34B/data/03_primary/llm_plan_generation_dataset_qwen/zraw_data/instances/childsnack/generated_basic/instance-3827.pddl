; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 908937

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child30 - child
    bread186 - bread-portion
    content57 - content-portion
    tray85 tray276 tray127 tray125 - tray
    table124 table5 table125 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray85 kitchen)
     (at tray276 kitchen)
     (at tray127 kitchen)
     (at tray125 kitchen)
     (at_kitchen_bread bread186)
     (at_kitchen_content content57)
     (not_allergic_gluten child30)
     (waiting child30 table125)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child30)
    )
  )
)
