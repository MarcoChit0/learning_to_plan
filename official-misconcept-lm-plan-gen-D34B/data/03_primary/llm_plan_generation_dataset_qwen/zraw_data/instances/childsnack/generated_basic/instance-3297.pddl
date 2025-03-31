; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 844520

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child492 child110 - child
    bread260 bread383 - bread-portion
    content335 content84 - content-portion
    tray65 tray446 tray449 tray165 - tray
    table70 table434 table21 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray65 kitchen)
     (at tray446 kitchen)
     (at tray449 kitchen)
     (at tray165 kitchen)
     (at_kitchen_bread bread260)
     (at_kitchen_bread bread383)
     (at_kitchen_content content335)
     (at_kitchen_content content84)
     (no_gluten_bread bread383)
     (no_gluten_content content335)
     (allergic_gluten child492)
     (not_allergic_gluten child110)
     (waiting child492 table70)
     (waiting child110 table21)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child492)
     (served child110)
    )
  )
)
