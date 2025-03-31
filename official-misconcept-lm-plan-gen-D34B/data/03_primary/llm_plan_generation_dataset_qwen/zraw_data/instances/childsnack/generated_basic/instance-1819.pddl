; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.9
; random seed: 281633

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child164 child391 - child
    bread202 bread32 - bread-portion
    content307 content30 - content-portion
    tray315 - tray
    table337 table476 table102 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray315 kitchen)
     (at_kitchen_bread bread202)
     (at_kitchen_bread bread32)
     (at_kitchen_content content307)
     (at_kitchen_content content30)
     (no_gluten_bread bread32)
     (no_gluten_content content307)
     (allergic_gluten child391)
     (not_allergic_gluten child164)
     (waiting child164 table102)
     (waiting child391 table337)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child164)
     (served child391)
    )
  )
)
