; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 193338

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child175 child121 - child
    bread77 bread214 - bread-portion
    content3 content202 - content-portion
    tray52 tray277 - tray
    table294 table249 table264 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray52 kitchen)
     (at tray277 kitchen)
     (at_kitchen_bread bread77)
     (at_kitchen_bread bread214)
     (at_kitchen_content content3)
     (at_kitchen_content content202)
     (not_allergic_gluten child175)
     (not_allergic_gluten child121)
     (waiting child175 table264)
     (waiting child121 table294)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child175)
     (served child121)
    )
  )
)
