; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.9
; random seed: 604169

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child419 child175 - child
    bread133 bread449 - bread-portion
    content320 content188 - content-portion
    tray126 tray223 tray304 tray87 - tray
    table439 table182 table255 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray126 kitchen)
     (at tray223 kitchen)
     (at tray304 kitchen)
     (at tray87 kitchen)
     (at_kitchen_bread bread133)
     (at_kitchen_bread bread449)
     (at_kitchen_content content320)
     (at_kitchen_content content188)
     (no_gluten_bread bread449)
     (no_gluten_content content188)
     (allergic_gluten child419)
     (not_allergic_gluten child175)
     (waiting child419 table182)
     (waiting child175 table255)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child419)
     (served child175)
    )
  )
)
