; child-snack task with 1 children and 0.7 gluten factor 
; constant factor of 1.3
; random seed: 939906

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child321 - child
    bread152 - bread-portion
    content179 - content-portion
    tray231 - tray
    table446 table206 table241 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray231 kitchen)
     (at_kitchen_bread bread152)
     (at_kitchen_content content179)
     (not_allergic_gluten child321)
     (waiting child321 table206)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child321)
    )
  )
)
