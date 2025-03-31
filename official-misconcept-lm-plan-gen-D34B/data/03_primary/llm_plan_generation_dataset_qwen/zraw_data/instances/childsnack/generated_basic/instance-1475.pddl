; child-snack task with 3 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 743291

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child447 child277 child373 - child
    bread21 bread236 bread38 - bread-portion
    content449 content452 content344 - content-portion
    tray105 - tray
    table22 table130 table462 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray105 kitchen)
     (at_kitchen_bread bread21)
     (at_kitchen_bread bread236)
     (at_kitchen_bread bread38)
     (at_kitchen_content content449)
     (at_kitchen_content content452)
     (at_kitchen_content content344)
     (no_gluten_bread bread236)
     (no_gluten_bread bread21)
     (no_gluten_content content449)
     (no_gluten_content content452)
     (allergic_gluten child447)
     (allergic_gluten child277)
     (not_allergic_gluten child373)
     (waiting child447 table130)
     (waiting child277 table462)
     (waiting child373 table22)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child447)
     (served child277)
     (served child373)
    )
  )
)
