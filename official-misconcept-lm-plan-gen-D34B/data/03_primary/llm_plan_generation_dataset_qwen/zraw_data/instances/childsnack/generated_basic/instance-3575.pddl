; child-snack task with 3 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 541737

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child37 child12 child396 - child
    bread264 bread380 bread444 - bread-portion
    content477 content406 content464 - content-portion
    tray483 tray401 tray200 - tray
    table62 table443 table467 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray483 kitchen)
     (at tray401 kitchen)
     (at tray200 kitchen)
     (at_kitchen_bread bread264)
     (at_kitchen_bread bread380)
     (at_kitchen_bread bread444)
     (at_kitchen_content content477)
     (at_kitchen_content content406)
     (at_kitchen_content content464)
     (no_gluten_bread bread264)
     (no_gluten_content content464)
     (allergic_gluten child396)
     (not_allergic_gluten child37)
     (not_allergic_gluten child12)
     (waiting child37 table443)
     (waiting child12 table467)
     (waiting child396 table443)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child37)
     (served child12)
     (served child396)
    )
  )
)
