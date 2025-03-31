; child-snack task with 2 children and 0.36 gluten factor 
; constant factor of 1.9
; random seed: 979909

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child353 child27 - child
    bread260 bread170 - bread-portion
    content383 content466 - content-portion
    tray241 tray115 - tray
    table193 table458 table102 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray241 kitchen)
     (at tray115 kitchen)
     (at_kitchen_bread bread260)
     (at_kitchen_bread bread170)
     (at_kitchen_content content383)
     (at_kitchen_content content466)
     (not_allergic_gluten child27)
     (not_allergic_gluten child353)
     (waiting child353 table458)
     (waiting child27 table458)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child353)
     (served child27)
    )
  )
)
