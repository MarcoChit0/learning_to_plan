; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 810563

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child5 child368 - child
    bread266 bread270 - bread-portion
    content47 content313 - content-portion
    tray449 tray29 - tray
    table497 table450 table145 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray449 kitchen)
     (at tray29 kitchen)
     (at_kitchen_bread bread266)
     (at_kitchen_bread bread270)
     (at_kitchen_content content47)
     (at_kitchen_content content313)
     (not_allergic_gluten child5)
     (not_allergic_gluten child368)
     (waiting child5 table497)
     (waiting child368 table497)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child5)
     (served child368)
    )
  )
)
