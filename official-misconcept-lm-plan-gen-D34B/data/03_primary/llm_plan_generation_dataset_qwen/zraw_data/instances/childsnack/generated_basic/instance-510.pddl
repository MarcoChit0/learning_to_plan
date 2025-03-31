; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 254851

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child356 child84 - child
    bread360 bread364 - bread-portion
    content286 content177 - content-portion
    tray7 tray470 - tray
    table479 table425 table370 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray7 kitchen)
     (at tray470 kitchen)
     (at_kitchen_bread bread360)
     (at_kitchen_bread bread364)
     (at_kitchen_content content286)
     (at_kitchen_content content177)
     (not_allergic_gluten child356)
     (not_allergic_gluten child84)
     (waiting child356 table425)
     (waiting child84 table370)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child356)
     (served child84)
    )
  )
)
