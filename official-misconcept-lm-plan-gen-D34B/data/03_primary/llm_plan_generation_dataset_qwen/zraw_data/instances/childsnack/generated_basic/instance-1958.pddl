; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 648610

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child422 child51 - child
    bread484 bread265 - bread-portion
    content0 content26 - content-portion
    tray241 - tray
    table264 table361 table74 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray241 kitchen)
     (at_kitchen_bread bread484)
     (at_kitchen_bread bread265)
     (at_kitchen_content content0)
     (at_kitchen_content content26)
     (no_gluten_bread bread265)
     (no_gluten_content content0)
     (allergic_gluten child422)
     (not_allergic_gluten child51)
     (waiting child422 table361)
     (waiting child51 table264)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child422)
     (served child51)
    )
  )
)
