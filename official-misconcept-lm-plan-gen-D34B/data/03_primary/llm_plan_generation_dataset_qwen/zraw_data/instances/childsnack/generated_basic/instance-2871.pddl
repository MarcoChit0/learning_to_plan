; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 881765

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child372 - child
    bread148 - bread-portion
    content364 - content-portion
    tray293 tray335 - tray
    table118 table379 table177 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray293 kitchen)
     (at tray335 kitchen)
     (at_kitchen_bread bread148)
     (at_kitchen_content content364)
     (not_allergic_gluten child372)
     (waiting child372 table177)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child372)
    )
  )
)
