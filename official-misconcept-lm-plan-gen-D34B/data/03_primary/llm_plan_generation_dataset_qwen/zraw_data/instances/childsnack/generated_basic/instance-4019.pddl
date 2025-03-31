; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 896007

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child417 child122 child12 - child
    bread315 bread347 bread279 - bread-portion
    content379 content165 content5 - content-portion
    tray492 - tray
    table96 table331 table335 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray492 kitchen)
     (at_kitchen_bread bread315)
     (at_kitchen_bread bread347)
     (at_kitchen_bread bread279)
     (at_kitchen_content content379)
     (at_kitchen_content content165)
     (at_kitchen_content content5)
     (not_allergic_gluten child12)
     (not_allergic_gluten child122)
     (not_allergic_gluten child417)
     (waiting child417 table335)
     (waiting child122 table331)
     (waiting child12 table335)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child417)
     (served child122)
     (served child12)
    )
  )
)
