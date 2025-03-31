; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 895290

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child11 child264 - child
    bread265 bread493 - bread-portion
    content243 content434 - content-portion
    tray398 tray125 - tray
    table12 table163 table419 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray398 kitchen)
     (at tray125 kitchen)
     (at_kitchen_bread bread265)
     (at_kitchen_bread bread493)
     (at_kitchen_content content243)
     (at_kitchen_content content434)
     (no_gluten_bread bread493)
     (no_gluten_content content243)
     (allergic_gluten child11)
     (not_allergic_gluten child264)
     (waiting child11 table419)
     (waiting child264 table163)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child11)
     (served child264)
    )
  )
)
