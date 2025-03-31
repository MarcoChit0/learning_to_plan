; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 950017

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child154 child115 - child
    bread96 bread25 - bread-portion
    content243 content350 - content-portion
    tray176 - tray
    table353 table191 table436 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray176 kitchen)
     (at_kitchen_bread bread96)
     (at_kitchen_bread bread25)
     (at_kitchen_content content243)
     (at_kitchen_content content350)
     (no_gluten_bread bread25)
     (no_gluten_content content350)
     (allergic_gluten child115)
     (not_allergic_gluten child154)
     (waiting child154 table353)
     (waiting child115 table353)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child154)
     (served child115)
    )
  )
)
