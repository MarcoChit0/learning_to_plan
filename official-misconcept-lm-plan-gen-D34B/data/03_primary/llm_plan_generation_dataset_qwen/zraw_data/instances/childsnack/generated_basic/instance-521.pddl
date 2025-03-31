; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 466653

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child179 child177 - child
    bread72 bread476 - bread-portion
    content104 content422 - content-portion
    tray70 tray333 - tray
    table60 table446 table338 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray70 kitchen)
     (at tray333 kitchen)
     (at_kitchen_bread bread72)
     (at_kitchen_bread bread476)
     (at_kitchen_content content104)
     (at_kitchen_content content422)
     (not_allergic_gluten child179)
     (not_allergic_gluten child177)
     (waiting child179 table338)
     (waiting child177 table60)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child179)
     (served child177)
    )
  )
)
