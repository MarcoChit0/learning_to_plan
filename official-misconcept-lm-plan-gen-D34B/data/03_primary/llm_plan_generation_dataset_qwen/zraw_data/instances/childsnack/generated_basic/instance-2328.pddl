; child-snack task with 1 children and 0.7 gluten factor 
; constant factor of 1.3
; random seed: 182789

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child428 - child
    bread108 - bread-portion
    content273 - content-portion
    tray29 - tray
    table378 table292 table170 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray29 kitchen)
     (at_kitchen_bread bread108)
     (at_kitchen_content content273)
     (not_allergic_gluten child428)
     (waiting child428 table292)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child428)
    )
  )
)
