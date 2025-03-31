; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 162305

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child244 child120 - child
    bread58 bread155 - bread-portion
    content95 content123 - content-portion
    tray439 tray25 - tray
    table426 table295 table235 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray439 kitchen)
     (at tray25 kitchen)
     (at_kitchen_bread bread58)
     (at_kitchen_bread bread155)
     (at_kitchen_content content95)
     (at_kitchen_content content123)
     (no_gluten_bread bread155)
     (no_gluten_content content123)
     (allergic_gluten child120)
     (not_allergic_gluten child244)
     (waiting child244 table235)
     (waiting child120 table426)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child244)
     (served child120)
    )
  )
)
