; child-snack task with 2 children and 0.28 gluten factor 
; constant factor of 1.7
; random seed: 333071

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child489 child349 - child
    bread409 bread174 - bread-portion
    content40 content139 - content-portion
    tray462 - tray
    table353 table22 table112 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray462 kitchen)
     (at_kitchen_bread bread409)
     (at_kitchen_bread bread174)
     (at_kitchen_content content40)
     (at_kitchen_content content139)
     (not_allergic_gluten child349)
     (not_allergic_gluten child489)
     (waiting child489 table22)
     (waiting child349 table22)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child489)
     (served child349)
    )
  )
)
