; child-snack task with 2 children and 0.31 gluten factor 
; constant factor of 1.7
; random seed: 222886

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child108 child416 - child
    bread415 bread136 - bread-portion
    content22 content310 - content-portion
    tray324 tray365 tray191 - tray
    table200 table463 table105 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray324 kitchen)
     (at tray365 kitchen)
     (at tray191 kitchen)
     (at_kitchen_bread bread415)
     (at_kitchen_bread bread136)
     (at_kitchen_content content22)
     (at_kitchen_content content310)
     (not_allergic_gluten child416)
     (not_allergic_gluten child108)
     (waiting child108 table200)
     (waiting child416 table463)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child108)
     (served child416)
    )
  )
)
