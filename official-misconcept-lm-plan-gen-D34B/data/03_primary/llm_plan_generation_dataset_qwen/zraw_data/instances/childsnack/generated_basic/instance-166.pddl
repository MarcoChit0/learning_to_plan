; child-snack task with 2 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 326803

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child299 child274 - child
    bread53 bread395 - bread-portion
    content477 content484 - content-portion
    tray444 tray194 tray35 tray275 - tray
    table292 table36 table235 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray444 kitchen)
     (at tray194 kitchen)
     (at tray35 kitchen)
     (at tray275 kitchen)
     (at_kitchen_bread bread53)
     (at_kitchen_bread bread395)
     (at_kitchen_content content477)
     (at_kitchen_content content484)
     (not_allergic_gluten child299)
     (not_allergic_gluten child274)
     (waiting child299 table36)
     (waiting child274 table292)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child299)
     (served child274)
    )
  )
)
