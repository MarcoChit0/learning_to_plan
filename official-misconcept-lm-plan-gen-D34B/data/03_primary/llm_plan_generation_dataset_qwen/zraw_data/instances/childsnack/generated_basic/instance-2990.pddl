; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 133407

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child134 child212 - child
    bread24 bread310 - bread-portion
    content123 content53 - content-portion
    tray122 tray491 tray349 tray300 - tray
    table459 table60 table235 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray122 kitchen)
     (at tray491 kitchen)
     (at tray349 kitchen)
     (at tray300 kitchen)
     (at_kitchen_bread bread24)
     (at_kitchen_bread bread310)
     (at_kitchen_content content123)
     (at_kitchen_content content53)
     (not_allergic_gluten child212)
     (not_allergic_gluten child134)
     (waiting child134 table235)
     (waiting child212 table235)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child134)
     (served child212)
    )
  )
)
