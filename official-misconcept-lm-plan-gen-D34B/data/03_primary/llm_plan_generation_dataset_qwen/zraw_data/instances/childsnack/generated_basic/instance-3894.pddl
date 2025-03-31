; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 855973

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child368 - child
    bread8 - bread-portion
    content199 - content-portion
    tray491 tray452 - tray
    table221 table406 table394 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray491 kitchen)
     (at tray452 kitchen)
     (at_kitchen_bread bread8)
     (at_kitchen_content content199)
     (not_allergic_gluten child368)
     (waiting child368 table406)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child368)
    )
  )
)
