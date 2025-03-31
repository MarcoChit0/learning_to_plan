; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.9
; random seed: 878637

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child265 child130 - child
    bread284 bread197 - bread-portion
    content179 content264 - content-portion
    tray469 - tray
    table236 table144 table459 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray469 kitchen)
     (at_kitchen_bread bread284)
     (at_kitchen_bread bread197)
     (at_kitchen_content content179)
     (at_kitchen_content content264)
     (no_gluten_bread bread284)
     (no_gluten_content content179)
     (allergic_gluten child265)
     (not_allergic_gluten child130)
     (waiting child265 table459)
     (waiting child130 table144)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child265)
     (served child130)
    )
  )
)
