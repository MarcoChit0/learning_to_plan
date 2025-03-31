; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 756729

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child169 child476 - child
    bread242 bread397 - bread-portion
    content279 content20 - content-portion
    tray191 tray30 - tray
    table463 table489 table464 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray191 kitchen)
     (at tray30 kitchen)
     (at_kitchen_bread bread242)
     (at_kitchen_bread bread397)
     (at_kitchen_content content279)
     (at_kitchen_content content20)
     (not_allergic_gluten child476)
     (not_allergic_gluten child169)
     (waiting child169 table489)
     (waiting child476 table489)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child169)
     (served child476)
    )
  )
)
