; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 232435

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child392 child327 - child
    bread284 bread422 - bread-portion
    content419 content461 - content-portion
    tray396 tray266 tray65 - tray
    table284 table315 table323 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray396 kitchen)
     (at tray266 kitchen)
     (at tray65 kitchen)
     (at_kitchen_bread bread284)
     (at_kitchen_bread bread422)
     (at_kitchen_content content419)
     (at_kitchen_content content461)
     (not_allergic_gluten child392)
     (not_allergic_gluten child327)
     (waiting child392 table323)
     (waiting child327 table315)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child392)
     (served child327)
    )
  )
)
