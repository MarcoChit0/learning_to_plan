; child-snack task with 2 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 973392

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child292 child119 - child
    bread369 bread255 - bread-portion
    content149 content23 - content-portion
    tray154 tray334 tray489 tray471 - tray
    table52 table44 table129 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray154 kitchen)
     (at tray334 kitchen)
     (at tray489 kitchen)
     (at tray471 kitchen)
     (at_kitchen_bread bread369)
     (at_kitchen_bread bread255)
     (at_kitchen_content content149)
     (at_kitchen_content content23)
     (no_gluten_bread bread255)
     (no_gluten_content content23)
     (allergic_gluten child292)
     (not_allergic_gluten child119)
     (waiting child292 table129)
     (waiting child119 table129)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child292)
     (served child119)
    )
  )
)
