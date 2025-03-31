; child-snack task with 2 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 287001

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child313 child339 - child
    bread452 bread369 - bread-portion
    content101 content20 - content-portion
    tray9 tray345 tray413 tray243 - tray
    table117 table264 table266 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray9 kitchen)
     (at tray345 kitchen)
     (at tray413 kitchen)
     (at tray243 kitchen)
     (at_kitchen_bread bread452)
     (at_kitchen_bread bread369)
     (at_kitchen_content content101)
     (at_kitchen_content content20)
     (no_gluten_bread bread369)
     (no_gluten_content content101)
     (allergic_gluten child313)
     (not_allergic_gluten child339)
     (waiting child313 table264)
     (waiting child339 table264)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child313)
     (served child339)
    )
  )
)
