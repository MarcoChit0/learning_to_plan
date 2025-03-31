; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 147714

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child193 child395 child305 - child
    bread149 bread78 bread233 - bread-portion
    content296 content307 content303 - content-portion
    tray47 - tray
    table405 table57 table455 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray47 kitchen)
     (at_kitchen_bread bread149)
     (at_kitchen_bread bread78)
     (at_kitchen_bread bread233)
     (at_kitchen_content content296)
     (at_kitchen_content content307)
     (at_kitchen_content content303)
     (not_allergic_gluten child193)
     (not_allergic_gluten child395)
     (not_allergic_gluten child305)
     (waiting child193 table405)
     (waiting child395 table57)
     (waiting child305 table405)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child193)
     (served child395)
     (served child305)
    )
  )
)
