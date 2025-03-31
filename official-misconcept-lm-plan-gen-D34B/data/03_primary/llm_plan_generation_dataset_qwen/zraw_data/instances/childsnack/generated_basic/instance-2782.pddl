; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 43026

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child334 child113 - child
    bread20 bread496 - bread-portion
    content249 content357 - content-portion
    tray341 - tray
    table303 table467 table81 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray341 kitchen)
     (at_kitchen_bread bread20)
     (at_kitchen_bread bread496)
     (at_kitchen_content content249)
     (at_kitchen_content content357)
     (no_gluten_bread bread20)
     (no_gluten_content content357)
     (allergic_gluten child334)
     (not_allergic_gluten child113)
     (waiting child334 table467)
     (waiting child113 table303)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child334)
     (served child113)
    )
  )
)
