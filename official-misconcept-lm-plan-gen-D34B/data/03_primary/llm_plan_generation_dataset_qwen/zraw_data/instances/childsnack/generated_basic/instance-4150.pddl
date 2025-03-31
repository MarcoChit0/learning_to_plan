; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 636008

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child32 child399 - child
    bread324 bread23 - bread-portion
    content321 content325 - content-portion
    tray156 tray315 tray42 - tray
    table182 table378 table392 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray156 kitchen)
     (at tray315 kitchen)
     (at tray42 kitchen)
     (at_kitchen_bread bread324)
     (at_kitchen_bread bread23)
     (at_kitchen_content content321)
     (at_kitchen_content content325)
     (not_allergic_gluten child399)
     (not_allergic_gluten child32)
     (waiting child32 table378)
     (waiting child399 table392)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child32)
     (served child399)
    )
  )
)
