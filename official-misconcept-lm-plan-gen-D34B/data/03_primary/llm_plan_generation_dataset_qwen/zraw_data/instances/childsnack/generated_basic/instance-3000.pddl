; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 798857

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child135 child12 - child
    bread119 bread375 - bread-portion
    content7 content263 - content-portion
    tray302 tray217 tray389 tray354 - tray
    table152 table302 table407 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray302 kitchen)
     (at tray217 kitchen)
     (at tray389 kitchen)
     (at tray354 kitchen)
     (at_kitchen_bread bread119)
     (at_kitchen_bread bread375)
     (at_kitchen_content content7)
     (at_kitchen_content content263)
     (not_allergic_gluten child12)
     (not_allergic_gluten child135)
     (waiting child135 table407)
     (waiting child12 table407)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child135)
     (served child12)
    )
  )
)
