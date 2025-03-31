; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 279630

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child350 - child
    bread422 - bread-portion
    content353 - content-portion
    tray367 tray55 - tray
    table219 table466 table152 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray367 kitchen)
     (at tray55 kitchen)
     (at_kitchen_bread bread422)
     (at_kitchen_content content353)
     (not_allergic_gluten child350)
     (waiting child350 table219)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child350)
    )
  )
)
