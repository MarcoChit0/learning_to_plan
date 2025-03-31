; child-snack task with 1 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 703887

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child349 - child
    bread45 - bread-portion
    content167 - content-portion
    tray322 - tray
    table381 table364 table83 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray322 kitchen)
     (at_kitchen_bread bread45)
     (at_kitchen_content content167)
     (not_allergic_gluten child349)
     (waiting child349 table83)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child349)
    )
  )
)
