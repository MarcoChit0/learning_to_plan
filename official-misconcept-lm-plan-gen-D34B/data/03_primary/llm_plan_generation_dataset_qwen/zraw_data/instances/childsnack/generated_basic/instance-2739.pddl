; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 527504

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child62 child52 - child
    bread131 bread264 - bread-portion
    content90 content343 - content-portion
    tray446 - tray
    table168 table495 table380 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray446 kitchen)
     (at_kitchen_bread bread131)
     (at_kitchen_bread bread264)
     (at_kitchen_content content90)
     (at_kitchen_content content343)
     (no_gluten_bread bread264)
     (no_gluten_content content343)
     (allergic_gluten child62)
     (not_allergic_gluten child52)
     (waiting child62 table495)
     (waiting child52 table495)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child62)
     (served child52)
    )
  )
)
