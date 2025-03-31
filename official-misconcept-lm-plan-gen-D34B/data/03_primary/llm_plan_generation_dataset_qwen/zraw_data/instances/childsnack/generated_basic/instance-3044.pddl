; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 473457

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child136 child411 - child
    bread33 bread122 - bread-portion
    content498 content372 - content-portion
    tray270 tray278 tray144 - tray
    table366 table111 table12 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray270 kitchen)
     (at tray278 kitchen)
     (at tray144 kitchen)
     (at_kitchen_bread bread33)
     (at_kitchen_bread bread122)
     (at_kitchen_content content498)
     (at_kitchen_content content372)
     (no_gluten_bread bread122)
     (no_gluten_content content498)
     (allergic_gluten child411)
     (not_allergic_gluten child136)
     (waiting child136 table111)
     (waiting child411 table366)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child136)
     (served child411)
    )
  )
)
